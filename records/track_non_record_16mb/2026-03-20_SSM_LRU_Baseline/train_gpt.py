"""
Autoresearch SSM pretraining script. Single-GPU, single-file.
Implements a Linear Recurrent Unit (LRU) based language model.
Cherry-picked and simplified from the GPT baseline for SSM parameter golf.
Usage: uv run train_ssm.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# SSM Model: Linear Recurrent Unit (LRU) based Language Model
# ---------------------------------------------------------------------------

@dataclass
class SSMConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    d_state: int = 64       # state dimension per channel for LRU
    d_inner_mult: int = 2   # expansion factor for SSM inner dimension
    mlp_mult: int = 4       # MLP expansion factor


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# LRU: Linear Recurrent Unit
# ---------------------------------------------------------------------------
# Implements: h_t = A * h_{t-1} + B * x_t; y_t = Re(C * h_t) + D * x_t
# Where A is diagonal complex, parameterized via log magnitude + phase.
# Uses parallel scan for efficient training.

def complex_log(x_real, x_imag):
    """Log of complex number (magnitude, phase)."""
    mag = torch.sqrt(x_real * x_real + x_imag * x_imag)
    phase = torch.atan2(x_imag, x_real)
    return torch.log(mag + 1e-8), phase


def lru_parallel_scan(log_a_real, log_a_imag, Bu_real, Bu_imag):
    """
    Parallel associative scan for LRU.
    log_a: (B, D, T) log of diagonal A entries (complex via real + imag)
    Bu: (B, D, T) input projections (complex via real + imag)
    Returns: h (B, D, T) hidden states (complex via real + imag)
    """
    # Cumulative sum of log_a gives log(prod(a)) for the scan
    # h_t = sum_{s<=t} (prod_{s<k<=t} a_k) * Bu_s
    # = sum_{s<=t} exp(sum_{s<k<=t} log(a_k)) * Bu_s

    T = log_a_real.shape[-1]

    # Compute cumulative log_a: cum_log_a[t] = sum_{k=0}^{t} log_a[k]
    cum_log_a_real = torch.cumsum(log_a_real, dim=-1)  # (B, D, T)
    cum_log_a_imag = torch.cumsum(log_a_imag, dim=-1)  # (B, D, T)

    # For computing h_t = sum_{s=0}^{t} exp(cum_log_a[t] - cum_log_a[s]) * Bu[s]
    # = exp(cum_log_a[t]) * sum_{s=0}^{t} exp(-cum_log_a[s]) * Bu[s]

    # Compute exp(-cum_log_a) * Bu, then cumsum, then multiply by exp(cum_log_a)
    neg_cum_real = -cum_log_a_real
    neg_cum_imag = -cum_log_a_imag

    # exp(neg_cum) * Bu (complex multiply)
    exp_neg_real = torch.exp(neg_cum_real) * torch.cos(neg_cum_imag)
    exp_neg_imag = torch.exp(neg_cum_real) * torch.sin(neg_cum_imag)

    # Complex multiply: (exp_neg) * Bu
    scaled_real = exp_neg_real * Bu_real - exp_neg_imag * Bu_imag
    scaled_imag = exp_neg_real * Bu_imag + exp_neg_imag * Bu_real

    # Cumulative sum
    cum_scaled_real = torch.cumsum(scaled_real, dim=-1)
    cum_scaled_imag = torch.cumsum(scaled_imag, dim=-1)

    # Multiply by exp(cum_log_a)
    exp_cum_real = torch.exp(cum_log_a_real) * torch.cos(cum_log_a_imag)
    exp_cum_imag = torch.exp(cum_log_a_real) * torch.sin(cum_log_a_imag)

    # Complex multiply: exp(cum_log_a) * cum_scaled
    h_real = exp_cum_real * cum_scaled_real - exp_cum_imag * cum_scaled_imag
    h_imag = exp_cum_real * cum_scaled_imag + exp_cum_imag * cum_scaled_real

    return h_real, h_imag


class LRU(nn.Module):
    """Linear Recurrent Unit with complex diagonal recurrence."""

    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is parameterized as exp(log_mag + i*phase) for stability
        # Initialize magnitudes close to 1 (slow decay) with uniform phases
        self.log_a_mag = nn.Parameter(torch.zeros(d_model, d_state))
        nn.init.uniform_(self.log_a_mag, -1.0, -0.01)  # magnitudes in (0.37, 0.99)

        self.a_phase = nn.Parameter(torch.zeros(d_model, d_state))
        nn.init.uniform_(self.a_phase, 0, 2 * math.pi)

        # B projection: real input -> complex state
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        # C projection: complex state -> real output
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        # D: direct feedthrough
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        x: (B, T, D) input sequence
        returns: (B, T, D) output sequence
        """
        B, T, D = x.shape

        # Compute A = exp(log_mag) * exp(i * phase) -> log_a = log_mag + i*phase
        log_a_real = self.log_a_mag  # (D, N)
        log_a_imag = self.a_phase    # (D, N)

        # Expand for batch and time: (B, D, T) by broadcasting
        # log_a is time-invariant: (1, D*N, 1) broadcast to (B, D*N, T)
        log_a_r = log_a_real.reshape(1, -1, 1).expand(B, -1, T)  # (B, D*N, T)
        log_a_i = log_a_imag.reshape(1, -1, 1).expand(B, -1, T)  # (B, D*N, T)

        # Bu = B * x: x is (B, T, D), B_re/B_im are (D, N)
        # We need Bu for each state dim: (B, D*N, T)
        x_t = x.transpose(1, 2)  # (B, D, T)

        # For each channel d, Bu[d,n,t] = B[d,n] * x[b,t,d]
        # Reshape: x_t is (B, D, T), B_re is (D, N)
        # Bu_re[b, d*N+n, t] = B_re[d, n] * x_t[b, d, t]
        B_re_exp = self.B_re.unsqueeze(0).unsqueeze(-1)  # (1, D, N, 1)
        B_im_exp = self.B_im.unsqueeze(0).unsqueeze(-1)  # (1, D, N, 1)
        x_exp = x_t.unsqueeze(2)  # (B, D, 1, T)

        Bu_real = (B_re_exp * x_exp).reshape(B, -1, T)  # (B, D*N, T)
        Bu_imag = (B_im_exp * x_exp).reshape(B, -1, T)  # (B, D*N, T)

        # Parallel scan
        h_real, h_imag = lru_parallel_scan(log_a_r, log_a_i, Bu_real, Bu_imag)
        # h: (B, D*N, T)

        # Output: y = Re(C * h) + D * x
        h_real = h_real.reshape(B, D, self.d_state, T)
        h_imag = h_imag.reshape(B, D, self.d_state, T)

        C_re = self.C_re.unsqueeze(0).unsqueeze(-1)  # (1, D, N, 1)
        C_im = self.C_im.unsqueeze(0).unsqueeze(-1)  # (1, D, N, 1)

        # Re(C * h) = C_re * h_real - C_im * h_imag, summed over state dim
        y = (C_re * h_real - C_im * h_imag).sum(dim=2)  # (B, D, T)
        y = y.transpose(1, 2)  # (B, T, D)

        # Add direct feedthrough
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x

        return y


class SSMBlock(nn.Module):
    """SSM block: LRU + gated MLP with pre-norm."""

    def __init__(self, config):
        super().__init__()
        d_inner = config.n_embd * config.d_inner_mult

        # Input projection (with gating)
        self.in_proj = nn.Linear(config.n_embd, d_inner, bias=False)
        self.gate_proj = nn.Linear(config.n_embd, d_inner, bias=False)

        # SSM core
        self.lru = LRU(d_inner, config.d_state)

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.n_embd, bias=False)

        # MLP
        self.mlp_fc = nn.Linear(config.n_embd, config.mlp_mult * config.n_embd, bias=False)
        self.mlp_proj = nn.Linear(config.mlp_mult * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, x0=None):
        # SSM branch with gating
        z = norm(x)
        u = self.in_proj(z)
        gate = torch.sigmoid(self.gate_proj(z))
        y = self.lru(u)
        y = gate * y
        x = x + self.out_proj(y)

        # MLP branch
        z = norm(x)
        x = x + self.mlp_proj(F.relu(self.mlp_fc(z)).square())

        return x


class SSMLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([SSMBlock(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # SSM blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            nn.init.uniform_(block.in_proj.weight, -s, s)
            nn.init.uniform_(block.gate_proj.weight, -s, s)
            nn.init.zeros_(block.out_proj.weight)
            nn.init.uniform_(block.mlp_fc.weight, -s, s)
            nn.init.zeros_(block.mlp_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # SSM scan cost: roughly 2 * d_inner * d_state * T per layer (much less than attention)
        d_inner = self.config.n_embd * self.config.d_inner_mult
        ssm_flops = self.config.n_layer * 2 * d_inner * self.config.d_state * self.config.sequence_len
        return 6 * (nparams - nparams_exclude) + ssm_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        ssm_blocks = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + lm_head + ssm_blocks + scalars
        return {
            'wte': wte, 'lm_head': lm_head,
            'ssm_blocks': ssm_blocks, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate SSM-specific params from projection matrices
        matrix_params = []
        ssm_params = []
        for block in self.transformer.h:
            # Projection matrices -> Muon
            matrix_params.extend([block.in_proj.weight, block.gate_proj.weight,
                                  block.out_proj.weight, block.mlp_fc.weight, block.mlp_proj.weight])
            # SSM-specific params -> Adam (structured, often complex-valued)
            ssm_params.extend([block.lru.log_a_mag, block.lru.a_phase,
                               block.lru.B_re, block.lru.B_im,
                               block.lru.C_re, block.lru.C_im, block.lru.D])

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ssm_params, lr=0.01 * dmodel_lr_scale, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Group matrix params by shape for Muon
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only) - same as GPT baseline
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 32       # model_dim = depth * ASPECT_RATIO (smaller for speed)
D_STATE = 2             # LRU state dimension (minimal for speed)
D_INNER_MULT = 1        # no expansion
MLP_MULT = 2            # 2x MLP

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 8
DEVICE_BATCH_SIZE = 64   # maximize throughput

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    # Round to multiple of 64 for efficiency
    model_dim = ((base_dim + 63) // 64) * 64
    return SSMConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_embd=model_dim,
        d_state=D_STATE, d_inner_mult=D_INNER_MULT, mlp_mult=MLP_MULT,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = SSMLanguageModel(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
